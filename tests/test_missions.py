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
from go2.executor import Executor, _blocked  # noqa: E402
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
        self.on_things = True                       # room_at(radius 0) on a thing's own spot finds no room
        self.on_furniture = set()                   # labels ranged by what they're on ("support", on a bed)
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

    def room_at(self, x, y, radius_m=0.):
        if radius_m == 0. and self.on_things and any(math.hypot(t[2] - x, t[3] - y) < .1 for t in self.things):
            return None                              # as on the real map: a thing's own spot is occupied
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
                source = "support" if label in self.on_furniture else "lidar"
                seen.append(Detection(None, label, color, math.hypot(dx, dy), bearing, c * dx + sn * dy,
                                      -sn * dx + c * dy, source, .9, clothing=clothing))
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


def mission(tool, args, time_limit=None):
    """The mission tool's command; time_limit (s) set directly, for tests quicker than the tool's least (10 s)."""
    commands = resolve(tool, args)
    if time_limit is not None:
        commands[0].params["time_limit"] = time_limit
    return commands


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

    def run_mission(self, tool, args, timeout=30, time_limit=None):
        before = len(self.events)
        result = self.executor.submit(mission(tool, args, time_limit), "g", 1)[0]
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

    def test_rooms_renumbered_as_the_map_grows_are_each_checked_once(self):
        split, listed = [0], self.house.rooms

        def rooms(fresh=False):                          # every split of the map numbers the rooms afresh
            split[0] += 1
            return [{**r, "region": 100 * split[0] + r["region"]} for r in listed()]

        def region_near(x, y, radius_m=1., among=None):
            return 100 * split[0] + list(ROOMS).index(room_of(x, y)) if room_of(x, y) else None
        self.executor.rooms, self.executor.room_regions.region_near = rooms, region_near
        task = self.run_mission("search", {"target": "umbrella"}, timeout=10)
        self.assertEqual(task.detail["mission"]["outcome"], "not_found", task.detail)
        self.assertEqual(sorted(self.house.walks), sorted(ROOMS), "each room once")

    def test_when_the_walk_up_finds_no_spot_it_closes_in_on_the_nearest_one_it_can_reach(self):
        def no_spot(task, cancel):                       # furniture all round the spots next to it
            raise _blocked(task, "no route to a free stand-off position near the object", math.inf, 0.)
        self.executor._approach = no_spot
        self.executor._stand_near = lambda x, y, within_m=1.6, **kw: (x - 1., y)
        task = self.run_mission("search", {"target": "teddy bear"})
        mission = task.detail["mission"]
        self.assertEqual((mission["outcome"], mission["room"]), ("found", "bedroom 2"), mission)
        self.assertLessEqual(math.hypot(self.robot.x - 4.5, self.robot.y + 3.5), missions.REACH_M)

    def test_when_the_walk_up_and_closing_in_both_fail_it_isnt_found(self):
        def no_spot(task, cancel):
            raise _blocked(task, "no route to a free stand-off position near the object", math.inf, 0.)
        self.executor._approach = no_spot
        self.executor._stand_near = lambda x, y, within_m=1.6, **kw: None
        self.house.things = [["teddy bear", "brown", 5.8, -4.8, None]] + self.house.things[1:]  # boxed into a corner
        walk = self.house._goto

        def not_near_it(task, cancel):                   # nowhere within 1.5 m of it can be walked to
            p = task.command.params
            if math.hypot(p["x"] - 5.8, p["y"] + 4.8) < 1.5:
                raise AdapterError("no route there")
            walk(task, cancel)
        self.executor._goto = not_near_it
        task = self.run_mission("search", {"target": "teddy bear"}, timeout=15)
        self.assertNotEqual(task.detail["mission"]["outcome"], "found", task.detail)

    def test_by_a_bed_it_goes_round_to_the_side_nearest_the_teddy_bear(self):
        self.house.on_furniture.add("teddy bear")       # ranged by the side of the bed under it
        walk_up = self.house._approach

        def bed_end(task, cancel):                       # the walk-up ends at the far end of the bed, 1.9 m off
            walk_up(task, cancel)
            self.house.put(4.5 - 1.9, -3.5, 0.)
        self.executor._approach = bed_end
        self.executor._past_support = lambda detection, pose, most_m=2.: (4.5, -3.5)
        self.executor._stand_near = lambda x, y, within_m=1.6, **kw: (x, y + 1.)   # the long side of the bed
        task = self.run_mission("search", {"target": "teddy bear"})
        mission = task.detail["mission"]
        self.assertEqual(mission["outcome"], "found", mission)
        self.assertAlmostEqual(math.hypot(self.robot.x - 4.5, self.robot.y + 3.5), 1., places=2)
        self.assertAlmostEqual(self.robot.yaw % (2 * math.pi), 3 * math.pi / 2, places=2, msg="facing the teddy bear")
        self.assertIn("next to it", mission["result"])

    def test_a_likely_room_gets_a_second_look_from_its_far_side(self):
        look = self.house.look

        def bed_edge_in_the_way():                       # from the room's middle, the bed's edge hides the teddy
            if math.hypot(self.robot.x - 3.5, self.robot.y + 3.) < .5:
                kept, self.house.things = self.house.things, [t for t in self.house.things if t[0] != "teddy bear"]
                try:
                    return look()
                finally:
                    self.house.things = kept
            return look()
        self.house.look = self.executor.look = bed_edge_in_the_way
        far = {"bedroom_1": (-5.5, -4.5), "bedroom_2": (2., -4.5)}
        self.executor._far_spot = lambda region, x, y, clearance_m=.45, **kw: far.get(room_of(x, y))
        task = self.run_mission("search", {"target": "teddy bear"})
        self.assertEqual(task.detail["mission"]["outcome"], "found", task.detail)
        self.assertEqual(self.house.walks, ["bedroom_1", "bedroom_1", "bedroom_2", "bedroom_2"],
                         "into each bedroom, then to its far side")

    def test_wedged_with_no_room_to_turn_it_backs_out_and_carries_on(self):
        turns, moves, turn = [0], [], self.house._turn_by

        def wedged_once(task, cancel, angle):
            turns[0] += 1
            if turns[0] == 1:
                raise _blocked(task, "too close to an obstacle or a drop to turn safely", 0., 0.)
            turn(task, cancel, angle)
        self.executor._turn_by = wedged_once
        self.executor._velocity = lambda task, p, cancel, leg=None: moves.append(p["vx"])
        task = self.run_mission("search", {"target": "umbrella"})
        self.assertEqual(task.detail["mission"]["outcome"], "not_found", task.detail)
        self.assertEqual(moves, [-.2, -.2], "backed straight out, 1 m")
        self.assertIn("I had no room to turn, so I backed out", self.said())

    def test_a_room_it_cant_get_into_isnt_reported_as_checked(self):
        tries, walk = [], self.house._goto

        def blocked_door(task, cancel):                 # bedroom 2's doorway is blocked
            p = task.command.params
            if room_of(p["x"], p["y"]) == "bedroom_2":
                tries.append("bedroom_2")
                raise AdapterError("no route into bedroom 2")
            walk(task, cancel)
        self.executor._goto = blocked_door
        task = self.run_mission("search", {"target": "teddy bear"})
        mission = task.detail["mission"]
        self.assertEqual(mission["outcome"], "blocked", mission)
        self.assertIn("couldn't get into bedroom 2", mission["result"])
        self.assertNotIn("bedroom 2", mission["checked"])
        self.assertFalse(any("checked bedroom 2" in text for text in self.said()), self.said())
        self.assertEqual(tries, ["bedroom_2", "bedroom_2"], "tried, then once more after the other rooms")

    def test_a_numbered_room_means_that_room_only(self):
        task = self.run_mission("search", {"target": "teddy bear", "where": "bedroom 1"})
        self.assertEqual(task.detail["mission"]["outcome"], "not_found", task.detail)
        self.assertEqual(self.house.walks, ["bedroom_1"], "not bedroom 2, where the teddy bear is")

    def test_a_search_limited_to_a_room_doesnt_go_to_one_remembered_elsewhere(self):
        self.executor.object_memory.remember("teddy bear", 4.5, -3.5, "brown")      # in bedroom 2
        task = self.run_mission("search", {"target": "teddy bear", "where": "the kitchen"})
        self.assertEqual(task.detail["mission"]["outcome"], "not_found", task.detail)
        self.assertEqual(self.house.walks, ["kitchen"])
        self.assertFalse(any("I remember seeing" in text for text in self.said()))

    def test_a_saved_place_it_cant_get_to_isnt_reported_as_searched(self):
        self.executor.places.remember("toy box", 4., -3., 0.)
        self.executor._goto = lambda task, cancel: (_ for _ in ()).throw(AdapterError("no route to the toy box"))
        task = self.run_mission("search", {"target": "teddy bear", "where": "toy box"})
        mission = task.detail["mission"]
        self.assertEqual(mission["outcome"], "blocked", mission)
        self.assertIn("couldn't get to the toy box", mission["result"])
        self.assertNotIn("isn't there", mission["result"])

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

    def test_asking_again_after_a_timeout_carries_on(self):
        # The resume window counts from when the mission stopped: here it times out 3 s after it began, later than
        # a 2 s window from its start (as search's default 600 s limit equals the 600 s window).
        original, missions.RESUME_S = missions.RESUME_S, 2.
        self.addCleanup(setattr, missions, "RESUME_S", original)
        self.house.walk_delay = 1.
        first = self.run_mission("search", {"target": "umbrella"}, time_limit=3)
        self.assertEqual(first.detail["mission"]["outcome"], "timed_out")
        checked = len(first.detail["mission"]["checked"])
        self.assertGreater(checked, 0)
        self.house.walk_delay = 0.
        walked = len(self.house.walks)
        task = self.run_mission("search", {"target": "umbrella"})
        self.assertTrue(any("carrying on the search" in text for text in self.said()), self.said())
        self.assertEqual(task.detail["mission"]["outcome"], "not_found")
        self.assertEqual(len(self.house.walks) - walked, len(ROOMS) - checked,
                         "rooms checked before the timeout aren't checked again")

    def test_time_limit_ends_the_search_and_says_so(self):
        self.house.walk_delay = 2.
        task = self.run_mission("search", {"target": "umbrella"}, time_limit=3)
        self.assertEqual(task.detail["mission"]["outcome"], "timed_out")
        self.assertIn("ask again to carry on", task.detail["mission"]["result"])


class SearchExploringTests(MissionTestCase):
    """Only the hallway is on the map at first: the search explores to find the rooms."""

    def test_a_room_that_can_never_be_reached_doesnt_keep_the_search_going(self):
        unreachable(self.house, "bathroom")
        task = self.run_mission("search", {"target": "umbrella"}, timeout=10)
        self.assertEqual(task.detail["mission"]["outcome"], "not_found", task.detail)

    def test_explores_to_find_rooms_then_searches_them(self):
        task = self.run_mission("search", {"target": "teddy bear"})
        self.assertEqual(task.detail["mission"]["outcome"], "found", task.detail)
        self.assertTrue(self.house.visited, "it explored")


class GuardTests(MissionTestCase):
    known = tuple(ROOMS)

    def test_alerts_with_a_snapshot_when_someone_comes_in_and_never_moves(self):
        self.executor.places.remember("front door", -4.5, 0., math.pi)
        result = self.executor.submit(mission("guard", {"place": "front door"}, 4), "g", 1)[0]
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

    def test_a_saved_place_faces_the_way_in_not_the_heading_it_was_saved_with(self):
        self.executor.places.remember("front door", -4.5, 0., math.pi / 2)        # saved facing the wall
        self.executor._way_in = lambda x, y: 0.                                   # the way in is along +x
        self.run_mission("guard", {"place": "front door"}, time_limit=1)
        self.assertAlmostEqual(math.cos(self.robot.yaw), 1., places=3, msg="it guards the way in, not the wall")

    def test_an_unknown_place_is_refused(self):
        task = self.run_mission("guard", {"place": "garage", "duration_s": 10})
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

    def test_rooms_are_read_as_go_to_place_reads_them(self):
        task = self.run_mission("patrol", {"rooms": ["the kitchen", "bedroom 1"]}, time_limit=600)  # one round
        self.assertEqual(task.status, "completed", task.detail)
        self.assertEqual(sorted(self.house.walks), ["bedroom_1", "kitchen"], "not every bedroom")
        task = self.run_mission("patrol", {"rooms": ["garage"]})
        self.assertEqual(task.status, "failed")
        self.assertIn("garage", task.detail["error"])

    def test_a_room_named_between_rounds_isnt_a_change(self):
        unnamed = {"kitchen": "room_7"}                  # the kitchen isn't named yet in round 1
        listed = self.house.rooms
        house_room_at = self.house.room_at                # a thing's own spot finds no room, as on the real map

        def room_at(x, y, radius_m=0.):
            room = house_room_at(x, y, radius_m)
            return unnamed.get(room, room)
        self.executor.room_at = room_at
        self.executor.rooms = lambda fresh=False: [{**r, "name": unnamed.get(r["name"], r["name"])} for r in listed()]
        command = mission("patrol", {}, 600)
        command[0].params["every"] = 2.
        self.assertEqual(self.executor.submit(command, "g", 1)[0]["status"], "queued")
        self.assertTrue(wait_for(lambda: any("round 1" in t for t in self.said()), 10))
        unnamed.clear()                                  # by round 2 it's the kitchen; nothing has moved
        self.assertTrue(wait_for(lambda: any("round 2" in t for t in self.said()), 15))
        self.assertIn("nothing has changed", next(t for t in self.said() if "round 2" in t))
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

    def test_a_spot_that_can_never_be_reached_gets_one_more_try_then_explore_finishes(self):
        tries = unreachable(self.house, "bathroom")
        task = self.run_mission("explore", {}, timeout=10)
        self.assertEqual(task.detail["mission"]["outcome"], "explored", task.detail)
        self.assertEqual(tries, ["bathroom", "bathroom"], "tried once, then once more at the end, then given up")
        self.assertEqual(sorted(self.house.visited), sorted(set(ROOMS) - {"hallway", "bathroom"}))


def unreachable(house, room):
    """Make the exploration spot in `room` impossible to reach (boxed in by furniture); the attempts on it."""
    tries, visit = [], house._visit_viewpoint

    def blocked_there(task, cancel):
        p = task.command.params
        if room_of(p["x"], p["y"]) == room:
            tries.append(room)
            raise AdapterError("no route there")
        visit(task, cancel)
    house.ex._visit_viewpoint = blocked_there
    return tries


class MissionToolTests(unittest.TestCase):
    def test_tools_take_their_arguments_and_refuse_others(self):
        self.assertEqual(resolve("search", {"target": "Teddy Bear"})[0].params,
                         {"mission": "search", "time_limit": 600, "target": "teddy bear", "where": None})
        self.assertEqual(resolve("patrol", {"rooms": ["kitchen"], "every_s": 120})[0].params["every"], 120)
        self.assertEqual(resolve("explore", None)[0].params["time_limit"], 1200)
        self.assertEqual(resolve("guard", {"duration_s": 10})[0].params["time_limit"], 10)
        for tool, args in (("search", {}), ("search", {"target": ""}), ("search", {"target": "x", "colour": 1}),
                           ("search", {"target": "x", "time_limit_s": 5000}), ("patrol", {"every_s": 10}),
                           ("patrol", {"rooms": "kitchen"}), ("guard", {"duration_s": -1}),
                           ("guard", {"duration_s": 0}), ("patrol", {"duration_s": .01}),
                           ("explore", {"time_limit_s": 9.9})):        # a moment would end the mission at once
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


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class SearchInTheSimulatorTests(unittest.TestCase):
    """#18 acceptance: in the demo house, with nothing remembered, "find the teddy bear" checks rooms in a sensible
    order, finds it, walks up to it and says which room it was in. On simulated time (simulation/evaluate_missions.py:
    the same physics, LiDAR, camera and executor), about 3 minutes."""

    @needs_camera
    def test_finds_the_teddy_bear_in_the_demo_house(self):
        import evaluate_missions
        import house
        demo = house.house_demo()
        task, state, _, _, _ = evaluate_missions.run(demo, "search", {"target": "teddy bear"}, 1200)
        mission = task.detail["mission"]
        self.assertEqual(mission["outcome"], "found", mission)
        self.assertEqual(mission["room"], "bedroom 2", mission["result"])
        teddy = next(o for o in demo["objects"] if o["kind"] == "teddy")
        self.assertLess(math.hypot(state.x - teddy["pos"][0], state.y - teddy["pos"][1]), evaluate_missions.REACHED_M)
        self.assertFalse(any("checked" in line and ("kitchen" in line or "bathroom" in line) for line in mission["log"]),
                         f"no unlikely room before the bedrooms: {mission['log']}")
        self.assertEqual(state.extra["collision_count"], 0)
        self.assertIsNone(state.fault)


@unittest.skipUnless(HAVE_SIM, "simulation requirements not installed")
class GuardInTheSimulatorTests(unittest.TestCase):
    """#18 acceptance: guarding the demo house's hallway, Go2 alerts within 3 s with a snapshot when a person walks
    in, and never moves toward them."""

    @needs_camera
    def test_alerts_within_3_s_when_a_person_walks_in_and_never_moves(self):
        from go2.adapters.sim import SimAdapter
        import house
        demo = {**house.house_demo(), "people": True}     # the walking people are only in houses that ask
        sim = SimAdapter(python=sys.executable, scene="house", headless=True, world=demo)
        sim.connect()
        self.addCleanup(sim.close)
        alerts, events = [], []
        executor = Executor(on_event=events.append, memory_path=tempfile.mkdtemp() + "/memory.json",
                            on_progress=lambda task, text, alert: alerts.append((time.monotonic(), text, alert))
                            if alert else None)
        self.addCleanup(executor.close)
        executor.attach(sim)
        start = sim.state()
        executor.places.remember("watch spot", start.x, start.y, start.yaw)   # the hallway, facing along it
        self.assertEqual(executor.submit(resolve("guard", {"place": "watch spot", "duration_s": 45}), "g", 1)[0]
                         ["status"], "queued")
        self.assertTrue(wait_for(lambda: executor.current is not None and any(
            "guarding" in line for line in executor.current.detail.get("mission", {}).get("log", [])), 30))
        settled = sim.state()
        sim.walker("walker_red", [[3.5, -3.], [3.5, 0.], [0., 0.]], speed=.6)  # out of bedroom 2, along the hallway
        came_in = None
        while came_in is None and not events:
            asked = time.monotonic()                     # when we asked: where it was then (or later)
            try:                                         # a busy laptop can answer slowly; don't fail on that
                where = sim._rpc("walkers", timeout=5.).get("walker_red")
            except AdapterError:
                continue
            if where and house.room_at(demo, *where[:2]) == "hallway" and where[1] > -.6:
                came_in = asked
            time.sleep(.05)
        self.assertIsNotNone(came_in, "the person never came into the hallway")
        self.assertTrue(wait_for(lambda: alerts, 10), "no alert")
        when, text, alert = alerts[0]
        self.assertLess(when - came_in, 3., f"alerted {when - came_in:.1f} s after the person came in")
        self.assertIn("a person", text)
        self.assertIsNotNone(executor.alert_image(alert["id"]))
        self.assertTrue(wait_for(lambda: events, 60))
        self.assertEqual(events[-1].detail["mission"]["outcome"], "done", events[-1].detail)
        end = sim.state()
        self.assertLess(math.hypot(end.x - settled.x, end.y - settled.y), .1, "it never moved")
        self.assertIsNone(end.fault)
        sim.walker("walker_red")


if __name__ == "__main__":
    unittest.main()
