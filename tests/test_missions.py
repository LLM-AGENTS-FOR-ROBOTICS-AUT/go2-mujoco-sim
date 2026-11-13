"""Missions (#18): search, guard, patrol and explore, their tools, and how they report.

Most tests run the missions' decisions in a stand-in house: walking, turning, looking and walking up to things are
replaced by moves in a plan of the demo house (the robot is put where it goes; the camera sees what's in the same
room, within 4 m and 60 degrees of straight ahead). The simulator tests at the end run them for real.
"""
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "simulation"))

from go2 import missions  # noqa: E402
from go2.adapters.base import AdapterError  # noqa: E402
from go2.agent import Agent, describe_outcome  # noqa: E402
from go2.executor import Executor  # noqa: E402
from go2.mock_model import MockModel  # noqa: E402
from go2.perception import Detection  # noqa: E402
from go2.tools import COMPACT_TOOLS, MISSIONS, TOOLS, ToolError, resolve  # noqa: E402
from tests.fakes import FakeRobot, needs_camera, wait_for  # noqa: E402

try:
    import mujoco  # noqa: F401
    import torch  # noqa: F401
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False

ROOMS = {"hallway": (-6., -1., 6., 1.), "living_room": (-6., 1., 0., 5.), "kitchen": (0., 1., 6., 5.),
         "bedroom_1": (-6., -5., -2., -1.), "bathroom": (-2., -5., 1., -1.), "bedroom_2": (1., -5., 6., -1.)}


def room_of(x, y):
    return next((name for name, (x0, y0, x1, y1) in ROOMS.items() if x0 <= x < x1 and y0 <= y < y1), None)


def middle(name):
    x0, y0, x1, y1 = ROOMS[name]
    return (x0 + x1) / 2, (y0 + y1) / 2


class StandInHouse:
    """The executor's moves and senses, replaced by a plan of the demo house (see the module docstring)."""

    def __init__(self, test, executor, robot, things, known=("hallway",)):
        self.ex, self.robot = executor, robot
        self.things = list(things)                  # [label, color, x, y, clothing]
        self.known = set(known)                     # rooms on the map so far
        self.views = [middle(name) for name in ROOMS if name not in known]
        self.visited, self.walks = [], []
        self.walk_delay = 0.
        for name in ("rooms", "_goto", "_visit_viewpoint", "_turn_by", "look", "_approach", "_sport",
                     "exploration_viewpoint", "room_at", "_way_in"):
            setattr(executor, name, getattr(self, name))
        self.put(-4.5, 0., 0.)

    def put(self, x, y, yaw):
        with self.robot.lock:
            self.robot.x, self.robot.y, self.robot.yaw = x, y, yaw

    # ---- the executor's methods ---------------------------------------------------------------------------
    def rooms(self, fresh=False):
        return [{"name": name, "region": i, "x": middle(name)[0], "y": middle(name)[1], "area_m2": 10.}
                for i, name in enumerate(ROOMS) if name in self.known]

    def room_at(self, x, y):
        return room_of(x, y)

    def _way_in(self, x, y):
        return None

    def _goto(self, task, cancel):
        p = task.command.params
        for _ in range(int(self.walk_delay / .05)):
            self.ex._check(cancel)
            time.sleep(.05)
        self.walks.append(room_of(p["x"], p["y"]))
        self.put(p["x"], p["y"], self.robot.yaw)

    def _visit_viewpoint(self, task, cancel):
        p = task.command.params
        self.visited.append(room_of(p["x"], p["y"]))
        self.known.add(room_of(p["x"], p["y"]))
        self.put(p["x"], p["y"], p["yaw"])

    def exploration_viewpoint(self, visited):
        left = [v for v in self.views if (*v, 0.) not in visited and room_of(*v) not in self.known]
        return (*left[0], 0.) if left else None

    def _turn_by(self, task, cancel, angle):
        self.ex._check(cancel)
        self.put(self.robot.x, self.robot.y, self.robot.yaw + angle)

    def _sport(self, task, cancel):
        self.robot.calls.append(("sport", task.command.params["name"]))

    def look(self):
        s = self.robot
        seen = []
        for label, color, x, y, clothing in self.things:
            dx, dy = x - s.x, y - s.y
            bearing = math.degrees(math.atan2(dy, dx) - s.yaw)
            bearing = (bearing + 180) % 360 - 180
            if room_of(x, y) == room_of(s.x, s.y) and math.hypot(dx, dy) < 4 and abs(bearing) < 60:
                c, sn = math.cos(s.yaw), math.sin(s.yaw)
                seen.append(Detection(None, label, color, math.hypot(dx, dy), bearing, c * dx + sn * dy,
                                      -sn * dx + c * dy, "lidar", .9, clothing=clothing))
        pose = (s.x, s.y, s.yaw)
        with self.ex.lock:
            self.ex._stable_ids(seen, pose)
            self.ex.last_detections = [(d, pose) for d in seen]
            self.ex._remember(seen, pose, save=False)
            self.ex.last_frame = np.zeros((4, 4, 3), dtype=np.uint8)
        return {"objects": [d.public() for d in seen]}

    def _approach(self, task, cancel):
        target = task.command.params["target"]
        match = self.ex._match(target)[0] or self.ex._recall(target)
        if match is None:
            raise AdapterError(f"I can't find a {target}")
        if isinstance(match[0], Detection):
            det, (x, y, yaw) = match
            ox = x + math.cos(yaw) * det.x_m - math.sin(yaw) * det.y_m
            oy = y + math.sin(yaw) * det.x_m + math.cos(yaw) * det.y_m
        else:
            _, ox, oy = match
        self.put(ox - .8, oy, 0.)
        self.look()


class MissionTestCase(unittest.TestCase):
    things = [["teddy bear", "brown", 4.5, -3.5, None], ["refrigerator", "white", 4., 4., None],
              ["sports ball", "red", -3., 3., None], ["backpack", "blue", -4., -3., None]]

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.events, self.progress = [], []
        self.robot = FakeRobot()
        self.robot.connect()
        self.addCleanup(self.robot.close)
        self.executor = Executor(on_event=self.events.append, memory_path=temp.name + "/memory.json",
                                 on_progress=lambda task, text, alert: self.progress.append((text, alert)))
        self.executor.attach(self.robot)
        self.addCleanup(self.executor.close)
        self.house = StandInHouse(self, self.executor, self.robot, [list(t) for t in self.things],
                                  known=getattr(self, "known", ("hallway",)))

    def run_mission(self, tool, args, timeout=30):
        before = len(self.events)
        result = self.executor.submit(resolve(tool, args), "g", 1)[0]
        self.assertEqual(result["status"], "queued", result)
        self.assertTrue(wait_for(lambda: len(self.events) > before, timeout), "the mission didn't end")
        return self.events[-1]

    def said(self):
        return [text for text, _ in self.progress]


class SearchTests(MissionTestCase):
    known = tuple(ROOMS)

    def test_finds_the_teddy_bear_checking_bedrooms_before_other_rooms(self):
        task = self.run_mission("search", {"target": "teddy bear"})
        mission = task.detail["mission"]
        self.assertEqual(task.status, "completed", task.detail)
        self.assertEqual((mission["outcome"], mission["room"]), ("found", "bedroom 2"))
        self.assertEqual(self.house.walks, ["bedroom_1", "bedroom_2"], "the nearest bedroom first, no other room")
        self.assertIn("checked bedroom 1: no teddy bear", self.said())
        self.assertIn("found the teddy bear in bedroom 2", self.said())
        self.assertIn("bedroom 2", describe_outcome(task, [task]))

    def test_goes_straight_to_where_it_was_seen_before(self):
        self.executor.object_memory.remember("teddy bear", 4.5, -3.5, "brown")
        task = self.run_mission("search", {"target": "teddy"})
        self.assertEqual(task.detail["mission"]["outcome"], "found")
        self.assertEqual(self.house.walks, [], "no room-by-room search")
        self.assertIn("I remember seeing a teddy", self.said()[0])

    def test_something_not_in_the_house_is_reported_after_every_room(self):
        task = self.run_mission("search", {"target": "umbrella"})
        mission = task.detail["mission"]
        self.assertEqual(mission["outcome"], "not_found")
        self.assertEqual(sorted(self.house.walks), sorted(ROOMS))
        self.assertIn("couldn't find the umbrella", mission["result"])

    def test_where_limits_the_search_to_that_room(self):
        task = self.run_mission("search", {"target": "teddy bear", "where": "kitchen"})
        self.assertEqual(task.detail["mission"]["outcome"], "not_found")
        self.assertEqual(self.house.walks, ["kitchen"])

    def test_stop_cancels_and_asking_again_carries_on(self):
        self.house.walk_delay = 1.
        self.executor.submit(resolve("search", {"target": "teddy bear"}), "g", 1)
        self.assertTrue(wait_for(lambda: "checked bedroom 1: no teddy bear" in self.said(), 5))
        time.sleep(.3)                                # on the way to bedroom 2
        began = time.monotonic()
        self.executor.stop("user")
        self.assertTrue(wait_for(lambda: self.events and self.events[-1].status == "stopped", 3))
        self.assertLess(time.monotonic() - began, 1.)
        self.house.walk_delay = 0.
        task = self.run_mission("search", {"target": "teddy bear"})
        self.assertEqual(task.detail["mission"]["outcome"], "found")
        self.assertTrue(any("carrying on the search" in text for text in self.said()))
        self.assertEqual(self.house.walks.count("bedroom_1"), 1, "bedroom 1 isn't checked twice")

    def test_time_limit_ends_the_search_and_says_so(self):
        self.house.walk_delay = 2.
        task = self.run_mission("search", {"target": "umbrella", "time_limit_s": 3})
        self.assertEqual(task.detail["mission"]["outcome"], "timed_out")
        self.assertIn("ask again to carry on", task.detail["mission"]["result"])


class SearchExploringTests(MissionTestCase):
    """Only the hallway is on the map at first: the search explores to find the rooms."""

    def test_explores_to_find_rooms_then_searches_them(self):
        task = self.run_mission("search", {"target": "teddy bear"})
        self.assertEqual(task.detail["mission"]["outcome"], "found", task.detail)
        self.assertTrue(self.house.visited, "it explored")


class GuardTests(MissionTestCase):
    known = tuple(ROOMS)

    def test_alerts_with_a_snapshot_when_someone_comes_in_and_never_moves(self):
        self.executor.places.remember("front door", -4.5, 0., math.pi)
        result = self.executor.submit(resolve("guard", {"place": "front door", "duration_s": 4}), "g", 1)[0]
        self.assertEqual(result["status"], "queued", result)
        self.assertTrue(wait_for(lambda: any("guarding" in t for t in self.said()), 5))
        moves = len([c for c in self.robot.calls if c[0] == "move"])
        position = (self.robot.x, self.robot.y)
        self.house.things.append(["person", None, -5.5, .2, "red"])        # someone comes in, 1 m in front of Go2
        began = time.monotonic()
        self.assertTrue(wait_for(lambda: any(alert for _, alert in self.progress), 5))
        self.assertLess(time.monotonic() - began, 3.)
        text, alert = next((t, a) for t, a in self.progress if a)
        self.assertIn("a person in red", text)
        self.assertIsNotNone(self.executor.alert_image(alert["id"]))
        self.assertTrue(wait_for(lambda: self.events, 10))
        self.assertEqual(self.events[-1].detail["mission"]["outcome"], "done")
        self.assertIn("1 alert", self.events[-1].detail["mission"]["result"])
        self.assertEqual(len([c for c in self.robot.calls if c[0] == "move"]), moves, "it never moved")
        self.assertEqual((self.robot.x, self.robot.y), position)
        self.assertIn(("sport", "sit"), self.robot.calls)

    def test_an_unknown_place_is_refused(self):
        task = self.run_mission("guard", {"place": "garage", "duration_s": 5})
        self.assertEqual(task.status, "failed")
        self.assertIn("garage", task.detail["error"])


class PatrolTests(MissionTestCase):
    known = ("living_room", "bedroom_1", "kitchen")

    def test_reports_a_moved_thing_a_gone_thing_and_a_person(self):
        command = resolve("patrol", {"duration_s": 600})[0]
        command.params["every"] = 2.                 # the tool's least is 60 s; quicker rounds for the test
        self.assertEqual(self.executor.submit([command], "g", 1)[0]["status"], "queued")
        self.assertTrue(wait_for(lambda: any("round 1" in t for t in self.said()), 10))
        self.assertIn("noted where everything is", next(t for t in self.said() if "round 1" in t))
        ball = next(t for t in self.house.things if t[0] == "sports ball")
        ball[2:4] = [4., 3.]                          # the ball goes to the kitchen
        self.house.things = [t for t in self.house.things if t[0] != "backpack"]
        self.house.things.append(["person", None, -4., 3., "blue"])
        self.assertTrue(wait_for(lambda: any("round 2" in t for t in self.said()), 15))
        report = next(t for t in self.said() if "round 2" in t)
        self.assertIn("the red sports ball moved from the living room to the kitchen", report)
        self.assertIn("the blue backpack is gone from bedroom 1", report)
        self.assertIn("a person in blue in the living room", report)
        self.executor.stop("user")
        self.assertTrue(wait_for(lambda: self.events and self.events[-1].status == "stopped", 3))


class ExploreTests(MissionTestCase):
    def test_visits_everywhere_and_lists_the_rooms(self):
        self.executor.rooms = lambda fresh=False: [
            {"name": name if name != "hallway" else "room_9", "region": i, "x": middle(name)[0], "y": middle(name)[1],
             "area_m2": 10.} for i, name in enumerate(ROOMS) if name in self.house.known]
        task = self.run_mission("explore", {})
        mission = task.detail["mission"]
        self.assertEqual(mission["outcome"], "explored")
        self.assertEqual(sorted(self.house.visited), sorted(set(ROOMS) - {"hallway"}))
        self.assertIn("6 rooms", mission["result"])
        self.assertIn("1 I couldn't name", mission["result"])
        self.assertIn("found the kitchen", self.said())


class MissionToolTests(unittest.TestCase):
    def test_tools_take_their_arguments_and_refuse_others(self):
        self.assertEqual(resolve("search", {"target": "Teddy Bear"})[0].params,
                         {"mission": "search", "time_limit": 300, "target": "teddy bear", "where": None})
        self.assertEqual(resolve("patrol", {"rooms": ["kitchen"], "every_s": 120})[0].params["every"], 120)
        self.assertEqual(resolve("explore", None)[0].params["time_limit"], 600)
        for tool, args in (("search", {}), ("search", {"target": ""}), ("search", {"target": "x", "colour": 1}),
                           ("search", {"target": "x", "time_limit_s": 5000}), ("patrol", {"every_s": 10}),
                           ("patrol", {"rooms": "kitchen"}), ("guard", {"duration_s": -1})):
            with self.subTest(tool=tool, args=args), self.assertRaises(ToolError):
                resolve(tool, args)

    def test_mission_tools_are_full_prompt_only_until_trained(self):
        names = {t["function"]["name"] for t in TOOLS}
        self.assertTrue(set(MISSIONS) <= names)
        self.assertFalse(set(MISSIONS) & {t["function"]["name"] for t in COMPACT_TOOLS})

    def test_the_real_go2_refuses_missions(self):
        robot = FakeRobot()
        robot.kind = "go2"
        robot.connect()
        self.addCleanup(robot.close)
        events = []
        executor = Executor(on_event=events.append)
        self.addCleanup(executor.close)
        executor.attach(robot)
        executor.submit(resolve("explore", {}), "g", 1)
        self.assertTrue(wait_for(lambda: events, 5))
        self.assertIn("simulator", events[-1].detail["error"])


class MockMissionTests(MissionTestCase):
    known = tuple(ROOMS)

    def test_mock_model_runs_a_search_and_reports_where_it_was(self):
        agent = Agent(MockModel(), self.executor)
        self.executor.on_progress = agent.on_mission_update
        self.executor.on_event = lambda task: (self.events.append(task), agent.on_task_event(task))
        agent.robot_line = lambda: "standing"
        self.addCleanup(agent.close)
        agent.submit("find the teddy bear")
        self.assertTrue(wait_for(lambda: self.events, 20))
        self.assertTrue(wait_for(lambda: not agent.busy and any("bedroom 2" in m["content"] for m in agent.messages
                                                               if m["role"] == "assistant"), 10))
        self.assertIn("search", [m["tool_name"] for m in agent.messages if m["role"] == "tool"])


if __name__ == "__main__":
    unittest.main()
