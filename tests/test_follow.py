"""Follow-me (#15): the follow tool, the person tracker, and the executor following people.

The executor tests use FakeRobot with FakePerson walkers in place of the camera (perception.fuse is replaced by
the people FakeRobot would see), so they run everywhere. The MuJoCo tests render the camera and are skipped
where MuJoCo can't render (GitHub's runners)."""
import math
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

from go2 import follow as F  # noqa: E402
from go2 import perception as P  # noqa: E402
from go2.agent import describe_outcome  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.tools import ToolError, resolve  # noqa: E402
from tests.fakes import FakePerson, FakeRobot, needs_camera, wait_for  # noqa: E402

try:
    import mujoco  # noqa: F401
    import torch  # noqa: F401
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False


class FollowToolTests(unittest.TestCase):
    def test_defaults(self):
        c = resolve("follow", {})[0]
        self.assertEqual(c.kind, "follow")
        self.assertEqual(c.params, {"target": "me", "color": None, "distance": 1.2, "duration": 60.})
        self.assertEqual(c.summary, "follow you at 1.2 m for up to 60 s")

    def test_clothing_distance_and_duration(self):
        c = resolve("follow", {"target": "the woman in the grey hoodie", "distance_m": 2, "duration_s": 90})[0]
        self.assertEqual((c.params["color"], c.params["distance"], c.params["duration"]), ("grey", 2., 90.))

    def test_limits_and_objects_are_refused(self):
        for args in ({"distance_m": .5}, {"distance_m": 4}, {"duration_s": 121}, {"target": "the red ball"},
                     {"target": "person in red and blue"}, {"speed_mps": .5}):
            with self.assertRaises(ToolError, msg=args):
                resolve("follow", args)


class TrackerTests(unittest.TestCase):
    def test_keeps_the_target_and_ignores_someone_far_from_it(self):
        t = F.PersonTracker(2., 0., 0., "red")
        for k in range(1, 11):                     # walking +x at 0.5 m/s, someone else standing at (2, 1.5)
            self.assertEqual(t.update(k * .2, [(2., 1.5, "blue"), (2. + .1 * k, 0., "red")]), 1)
        self.assertAlmostEqual(t.vx, .5, delta=.15)
        x, _ = t.predict(2.4)
        self.assertAlmostEqual(x, 3.2, delta=.15)

    def test_never_switches_to_a_different_shirt_even_when_closer(self):
        t = F.PersonTracker(2., 0., 0., "red")
        self.assertIsNone(t.update(.2, [(2.05, .05, "blue")]), "a blue shirt is not the red-shirted target")
        self.assertEqual(t.update(.4, [(2.05, .05, "blue"), (2.3, .1, "red")]), 1)

    def test_holds_when_two_unknowns_are_equally_likely(self):
        t = F.PersonTracker(2., 0., 0., None)
        self.assertIsNone(t.update(.2, [(2.1, .15, None), (2.1, -.15, None)]))
        self.assertTrue(t.ambiguous)
        self.assertEqual(t.update(.4, [(2.1, .05, None), (2.1, -.9, None)]), 0, "picks again once they separate")
        self.assertFalse(t.ambiguous)

    def test_gate_widens_while_unseen(self):
        t = F.PersonTracker(0., 0., 0., "red")
        self.assertIsNone(t.update(.2, [(1.2, 0., "red")]), "1.2 m jump is not the same person a frame later")
        self.assertEqual(t.update(2.0, [(1.2, 0., "red")]), 0, "but can be after two seconds out of sight")

    def test_speed_cap_matches_the_executor_margin(self):
        for clear in (.3, .5, 1., 2.):
            v = F.max_safe_speed(clear)
            self.assertLessEqual(Executor._margin(v), clear + 1e-9)
        self.assertEqual(F.max_safe_speed(.2), 0.)
        self.assertEqual(F.max_safe_speed(10.), .8)


class FollowExecutorTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.executor = Executor(on_event=self.events.append)
        self.robot = None
        self.patch = None

    def tearDown(self):
        self.executor.close()
        if self.robot:
            self.robot.close()
        if self.patch:
            self.patch.stop()

    def attach(self, people, **kwargs):
        self.robot = FakeRobot(people=people, camera=True, **kwargs)
        self.robot.connect()
        self.patch = mock.patch.object(P, "fuse", lambda *a, **k: self.robot.seen_people())
        self.patch.start()
        self.executor.attach(self.robot)
        return self.robot

    def follow(self, **args):
        return self.executor.submit(resolve("follow", args), "g1", 1)

    def finished(self, timeout=40):
        self.assertTrue(wait_for(lambda: self.events, timeout), "follow did not end")
        return self.events[-1]

    def gap(self, person):
        x, y = person.at(time.monotonic())
        return math.hypot(x - self.robot.x, y - self.robot.y) - .15 - P.HALF_LENGTH

    def test_follows_a_walking_person_at_the_set_distance(self):
        person = FakePerson([(2, 0), (6, 0), (6, 3)], speed=.5, delay=1.)
        self.attach([person])
        result = self.follow(duration_s=16)
        self.assertEqual(result[0]["status"], "queued")
        gaps = []
        deadline = time.monotonic() + 18
        while not self.events and time.monotonic() < deadline:
            time.sleep(.25)
            if self.executor.snapshot()["current"]:
                gaps.append(self.gap(person))
        task = self.finished()
        self.assertEqual(task.status, "completed", task.detail)
        settled = gaps[len(gaps) // 3:]
        self.assertLess(max(abs(g - 1.2) for g in settled), .4, f"gaps {[round(g, 2) for g in settled]}")
        self.assertGreater(task.detail["walked_m"], 4)
        self.assertIn("followed for", describe_outcome(task, [task]))

    def test_stop_ends_following_within_a_second(self):
        self.attach([FakePerson([(2, 0), (8, 0)], speed=.5)])
        self.follow()
        self.assertTrue(wait_for(lambda: (self.executor.snapshot()["current"] or {}).get("following"), 5))
        time.sleep(2)
        asked = time.monotonic()
        self.executor.stop("user")
        self.assertTrue(wait_for(lambda: abs(self.robot.v[0]) < .05 and abs(self.robot.v[2]) < .05, 2))
        self.assertLess(time.monotonic() - asked, 1.)
        self.assertEqual(self.finished().status, "stopped")

    def test_losing_the_person_stops_then_gives_up(self):
        person = FakePerson([(2, 0), (8, 0)], speed=.5)
        self.attach([person], obstacles=[(3.6, 3.9, 1.6, 1.9)])       # a landmark the robot can name
        self.follow()
        self.assertTrue(wait_for(lambda: (self.executor.snapshot()["current"] or {}).get("following"), 5))
        time.sleep(3)
        person.hide()
        hidden = time.monotonic()
        self.assertTrue(wait_for(lambda: abs(self.robot.cmd[0]) < 1e-6, F.SEEN_S + 1.5), "stopped walking")
        self.assertLess(time.monotonic() - hidden, 2.5)
        task = self.finished(F.GIVE_UP_S + 5)
        self.assertEqual(task.status, "failed")
        self.assertIn("lost sight of you", task.detail["error"])
        self.assertLess(task.finished - hidden, F.GIVE_UP_S + 1.5)

    def test_gets_the_person_back_after_a_short_disappearance(self):
        person = FakePerson([(2, 0), (8, 0)], speed=.4)
        self.attach([person])
        self.follow(duration_s=12)
        self.assertTrue(wait_for(lambda: (self.executor.snapshot()["current"] or {}).get("following"), 5))
        time.sleep(2)
        person.hidden = True
        time.sleep(1.5)
        person.hidden = False
        task = self.finished()
        self.assertEqual(task.status, "completed", task.detail)
        self.assertGreaterEqual(task.detail["lost_sight"], 1)
        self.assertLess(abs(self.gap(person) - 1.2), .5)

    def test_keeps_following_the_right_person_when_another_walks_past(self):
        red = FakePerson([(2, 0), (7, 0)], speed=.4, colour="red")
        blue = FakePerson([(4.5, 2.5), (2.5, -1.5)], speed=.6, colour="blue", delay=1.)   # cuts across in between
        self.attach([red, blue])
        self.follow(target="the person in red", duration_s=12)
        task = self.finished()
        self.assertEqual(task.status, "completed", task.detail)
        self.assertLess(abs(self.robot.y), .4, "still behind the red-shirted person, not drawn off by the blue one")
        self.assertLess(abs(self.gap(red) - 1.2), .5)

    def test_asks_which_person_when_two_are_equally_near(self):
        self.attach([FakePerson([(2, .8)], colour="red"), FakePerson([(2, -.8)], colour="blue")])
        result = self.follow()
        self.assertEqual(result[0]["status"], "failed")
        self.assertIn("can't tell which one", result[0]["error"])
        self.assertIn("in red", result[0]["error"])
        self.assertLess(abs(self.robot.x), .05, "did not move")

    def test_clothing_picks_one_of_two(self):
        red, blue = FakePerson([(2, .8)], colour="red"), FakePerson([(2, -.8)], colour="blue")
        self.attach([red, blue])
        self.follow(target="the person in blue", duration_s=4)
        task = self.finished()
        self.assertEqual(task.status, "completed")
        self.assertEqual(task.detail["target"]["clothing"], "blue")
        self.assertLess(self.robot.yaw, -.2, "turned toward the blue one on the right")

    def test_looks_around_for_someone_behind(self):
        self.attach([FakePerson([(-2.5, 0)], colour="red")])
        self.follow(duration_s=3)
        task = self.finished(30)
        self.assertEqual(task.status, "completed", task.detail)
        self.assertGreater(abs(self.robot.yaw), 2.5, "turned around to find them")

    def test_nobody_in_sight(self):
        self.attach([])
        self.follow()
        task = self.finished(30)
        self.assertEqual(task.status, "failed")
        self.assertIn("can't see anyone to follow", task.detail["error"])

    def test_an_obstacle_in_the_way_ends_it_as_blocked(self):
        self.attach([FakePerson([(2.5, 0), (7, 0)], speed=.5, delay=1.)], obstacles=[(1.4, 1.7, -.6, .6)])
        self.follow()
        task = self.finished(20)
        self.assertEqual(task.status, "blocked", task.detail)
        self.assertLess(self.robot.x + P.HALF_LENGTH, 1.4, "never reached the obstacle")

    def test_never_walks_into_the_person(self):
        person = FakePerson([(3, 0), (1.0, 0)], speed=.6)          # walks toward the robot and stops close
        self.attach([person])
        self.follow(duration_s=8)
        closest = math.inf
        while not self.events:
            closest = min(closest, self.gap(person))
            time.sleep(.05)
        self.assertGreater(closest, .2)

    def test_a_camera_faster_than_the_lidar_doesnt_make_it_trail(self):
        # People are placed by the LiDAR, refreshed 10 times a second; with frames every 10 ms most frames
        # repeat the last scan. Those repeats must not drag the speed estimate down.
        person = FakePerson([(2, 0), (12, 0)], speed=.5, delay=1.)
        robot = FakeRobot(people=[person], camera=True)
        robot.camera_period = .01
        robot.connect()
        self.robot = robot
        def fused(*a, **k):
            scans = self.executor.cloud.scans
            return robot.seen_people(at=scans[-1][0] if scans else None)
        self.patch = mock.patch.object(P, "fuse", fused)
        self.patch.start()
        self.executor.attach(robot)
        self.follow(duration_s=14)
        gaps, deadline = [], time.monotonic() + 16
        while not self.events and time.monotonic() < deadline:
            time.sleep(.25)
            gaps.append(self.gap(person))
        task = self.finished()
        self.assertEqual(task.status, "completed", task.detail)
        settled = gaps[len(gaps) // 2:]
        self.assertLess(max(abs(g - 1.2) for g in settled), .4, f"gaps {[round(g, 2) for g in settled]}")

    def test_glass_between_them_is_caught_by_the_stall_check(self):
        person = FakePerson([(2.5, 0), (8, 0)], speed=.5, delay=.5)
        robot = self.attach([person], glass=[(1.6, 1.65, -1, 1)])
        self.follow()
        task = self.finished(25)
        self.assertEqual(task.status, "blocked", task.detail)
        self.assertTrue(task.detail["blocked"].get("stall"))
        self.assertIsNotNone(robot.contact_at)
        self.assertLess(robot.x + .36, 1.6 - .05, "backed away from the glass")
        self.assertEqual(len(self.executor.bumps), 1)

    def test_doesnt_back_into_space_it_cant_see(self):
        person = FakePerson([(3, 0), (.9, 0)], speed=.6)          # walks up close to the robot
        robot = self.attach([person], blind_behind=True)
        self.follow(duration_s=7)
        lowest = 0.
        while not self.events:
            lowest = min(lowest, robot.x)
            time.sleep(.05)
        self.assertGreater(lowest, -.03, "never reversed into the unseen space behind it")

    def test_needs_a_camera(self):
        self.robot = FakeRobot()
        self.robot.connect()
        self.executor.attach(self.robot)
        self.follow()
        task = self.finished(15)
        self.assertEqual(task.status, "failed")
        self.assertIn("need the camera", task.detail["error"])

    def test_real_robot_is_refused_for_now(self):
        robot = self.attach([FakePerson([(2, 0)])])
        robot.kind = "go2"
        self.follow()
        task = self.finished(5)
        self.assertEqual(task.status, "failed")
        self.assertIn("only works in the simulator", task.detail["error"])


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class FollowSimulationTests(unittest.TestCase):
    """The MuJoCo room: a walking person (world.WALKERS) seen through the rendered camera and the LiDAR."""

    @classmethod
    def setUpClass(cls):
        from go2.adapters.sim import SimAdapter
        cls.sims = {}
        cls.SimAdapter = SimAdapter

    @classmethod
    def tearDownClass(cls):
        for sim in cls.sims.values():
            sim.close()

    def sim(self, scene):
        if scene not in self.sims:
            self.sims[scene] = self.SimAdapter(python=sys.executable, scene=scene, headless=True)
            self.sims[scene].connect()
        sim = self.sims[scene]
        sim.reset(0, 0, 0)
        self.events = []
        self.executor = Executor(on_event=self.events.append)
        self.executor.attach(sim)
        self.addCleanup(self.executor.close)
        return sim

    def true_gap(self, sim, name="walker_red"):
        x, y = sim.walkers()[name]
        s = sim.state()
        return math.hypot(x - s.x, y - s.y) - P.HALF_LENGTH - .12       # torso half-depth

    @needs_camera
    def test_follows_a_walking_person_for_a_minute(self):
        sim = self.sim("empty")
        sim.walker("walker_red", [[2, 0], [6, 0], [6, 4], [2, 4]], speed=.5, loop=True)
        self.executor.submit(resolve("follow", {"duration_s": 60}), "g", 1)
        gaps, began = [], time.monotonic()
        while not self.events and time.monotonic() - began < 75:
            time.sleep(.5)
            if time.monotonic() - began > 8:
                gaps.append(self.true_gap(sim))
        task = self.events[-1]
        self.assertEqual(task.status, "completed", task.detail)
        self.assertLess(max(abs(g - 1.2) for g in gaps), .4, f"gaps {min(gaps):.2f}..{max(gaps):.2f}")
        self.assertIsNone(sim.state().fault, "no collision or fall")
        sim.walker("walker_red")

    @needs_camera
    def test_follows_through_the_furnished_room(self):
        # The room scene, with the standing person in green further off: a clear path east past the furniture.
        sim = self.sim("room")
        sim.reset(-2.5, 0, 0)
        sim.walker("walker_red", [[-1.1, 0], [.4, 0], [.4, .7]], speed=.4)
        self.executor.submit(resolve("follow", {"duration_s": 20}), "g", 1)
        gaps, began = [], time.monotonic()
        while not self.events and time.monotonic() - began < 30:
            time.sleep(.5)
            if time.monotonic() - began > 8:
                gaps.append(self.true_gap(sim))
        task = self.events[-1]
        self.assertEqual(task.status, "completed", task.detail)
        self.assertLess(max(abs(g - 1.2) for g in gaps), .4, f"gaps {min(gaps):.2f}..{max(gaps):.2f}")
        self.assertIsNone(sim.state().fault, "no collision or fall")
        sim.walker("walker_red")

    @needs_camera
    def test_keeps_the_red_person_when_the_blue_one_walks_past_and_stops_when_they_vanish(self):
        sim = self.sim("empty")
        sim.walker("walker_red", [[2, 0], [7, 0]], speed=.4)
        sim.walker("walker_blue", [[4.5, 2.5], [3.0, -2.0]], speed=.5)
        self.executor.submit(resolve("follow", {"target": "the person in red", "duration_s": 40}), "g", 1)
        time.sleep(10)
        self.assertLess(abs(sim.state().y), .4, "behind the red one")
        sim.walker("walker_red")                                    # gone
        gone = time.monotonic()
        self.assertTrue(wait_for(lambda: sim.state().speed < .1, F.SEEN_S + 2.5))
        self.assertLess(time.monotonic() - gone, 3.)
        self.assertTrue(wait_for(lambda: self.events, F.GIVE_UP_S + 6))
        self.assertEqual(self.events[-1].status, "failed")
        self.assertIn("lost sight", self.events[-1].detail["error"])
        sim.walker("walker_blue")


if __name__ == "__main__":
    unittest.main()
