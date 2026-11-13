import json
import math
from pathlib import Path
import tempfile
import unittest

from go2.agent import Agent
from go2.executor import Executor
from go2.mock_model import MockModel
from go2.tools import ToolError, resolve
from go2 import perception as P
from tests.fakes import FakeRobot, wait_for


class PlaceToolTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.robot = FakeRobot()
        self.robot.connect()
        self.addCleanup(self.robot.close)
        self.events = []
        self.path = Path(self.folder.name) / "scene.json"
        self.executor = Executor(on_event=self.events.append, memory_path=self.path)
        self.executor.attach(self.robot)
        self.addCleanup(self.executor.close)

    def call(self, name, args):
        return self.executor.submit(resolve(name, args), "test", 1)[0]

    def test_remember_and_list_are_queries_and_survive_restart(self):
        with self.robot.lock:
            self.robot.x, self.robot.y, self.robot.yaw = 1.2, -.8, .4
        result = self.call("remember_place", {"name": "Door"})
        self.assertEqual(result["status"], "done")
        self.assertEqual(result["place"]["frame"], "map")
        self.assertTrue(result["place"]["persisted"])
        self.assertFalse(any(c[0] == "move" for c in self.robot.calls))
        restarted = Executor(memory_path=self.path)
        self.addCleanup(restarted.close)
        restarted.attach(self.robot)
        self.assertEqual(restarted.places.get("door").x, 1.2)
        listed = restarted.submit(resolve("list_places", {}), "test", 1)[0]
        self.assertEqual(listed["places"][0]["name"], "Door")

    def test_return_uses_saved_world_position_after_robot_turns(self):
        self.executor.places.remember("door", .6, .4, 1.)
        with self.robot.lock:
            self.robot.x, self.robot.y, self.robot.yaw = -.4, .1, math.pi
        result = self.call("go_to_place", {"name": "DOOR"})
        self.assertEqual(result["status"], "queued", result)
        self.assertTrue(wait_for(lambda: self.events, 20))
        task = self.events[-1]
        self.assertEqual(task.status, "completed", task.detail)
        self.assertLess(math.hypot(self.robot.x-.6, self.robot.y-.4), .2)
        self.assertEqual(task.detail["destination"]["x_m"], .6)

    def test_unknown_place_and_out_of_range_target_do_not_walk(self):
        result = self.call("go_to_place", {"name": "unknown"})
        self.assertEqual(result["status"], "rejected")
        self.executor.places.remember("far away", 21., 0., 0.)
        result = self.call("go_to_place", {"name": "far away"})
        self.assertTrue(wait_for(lambda: self.events, 3))
        self.assertIn("20 m", self.events[-1].detail["error"])
        self.assertFalse(any(c[0] == "move" for c in self.robot.calls))

    def test_memory_query_is_bounded_and_does_not_look_or_move(self):
        for i in range(8):
            self.executor.object_memory.remember("sports ball", i*2., 1., "red", save=False)
        self.executor.places.remember("front door", 2., 3., 0.)
        result = self.call("query_memory", {"target": "red ball", "limit": 2})
        self.assertEqual(len(result["objects"]), 2)
        self.assertEqual(result["objects_total"], 8)
        self.assertEqual(self.call("query_memory", {"target": "unicorn"})["objects"], [])
        self.assertEqual(self.call("query_memory", {"target": "front door"})["places"][0]["name"], "front door")
        self.assertFalse(any(c[0] == "move" for c in self.robot.calls))
        self.assertIsNone(self.executor.last_frame)

    def test_invalid_tool_names_and_limits_are_rejected(self):
        for name in ("remember_place", "go_to_place"):
            for args in ({}, {"name": ""}, {"name": "door", "x_m": 9}):
                with self.subTest(tool=name, args=args), self.assertRaises(ToolError):
                    resolve(name, args)
        for limit in (True, 0, 2.5, 11):
            with self.subTest(limit=limit), self.assertRaises(ToolError):
                resolve("query_memory", {"limit": limit})

    def test_mock_remember_then_go_resolves_place_from_executor(self):
        agent = Agent(MockModel(), self.executor)
        agent.robot_line = lambda: "standing at the current location"
        self.addCleanup(agent.close)
        agent.submit("remember this spot as the door")
        self.assertTrue(wait_for(lambda: not agent.busy, 3))
        self.assertEqual(self.executor.places.get("door").x, 0.)
        agent.submit("go to the door")
        self.assertTrue(wait_for(lambda: not agent.busy, 3))
        calls = [m["tool_name"] for m in agent.messages if m["role"] == "tool"]
        self.assertEqual(calls[-2:], ["query_memory", "go_to_place"])

    def test_mock_reports_missing_memory_without_issuing_new_motion(self):
        key = self.executor.object_memory.remember("laptop", 2., 0.)["id"]
        for stamp in (0., .2, .4):
            self.executor.object_memory.observe_empty([key], observed_at=stamp)
        agent = Agent(MockModel(), self.executor)
        self.addCleanup(agent.close)
        agent.robot_line = lambda: "standing"
        agent.submit("where is the laptop?")
        self.assertTrue(wait_for(lambda: not agent.busy, 3))
        self.assertIn("may have moved", agent.messages[-1]["content"])
        self.assertFalse(any(c[0] == "move" for c in self.robot.calls))

    def test_remembered_object_after_restart_is_revisited_without_searching(self):
        self.executor.object_memory.remember("laptop", 2., 0.)
        self.executor.close()
        self.executor = Executor(on_event=self.events.append, memory_path=self.path)
        self.addCleanup(self.executor.close)
        self.robot.yaw = math.pi
        self.executor.attach(self.robot)
        turns = []
        original_turn = self.executor._turn_by
        def turn(task, cancel, angle):
            turns.append(angle)
            return original_turn(task, cancel, angle)
        self.executor._turn_by = turn
        def look():
            self.executor._refresh_cloud()
            state = self.robot.state()
            dx, dy = 2.-state.x, -state.y
            c, s = math.cos(state.yaw), math.sin(state.yaw)
            x, y = c*dx+s*dy, -s*dx+c*dy
            bearing = math.degrees(math.atan2(y, x))
            detections = [] if abs(bearing) > 60 else [P.Detection("laptop1", "laptop", None,
                            math.hypot(x,y), bearing, x, y, "lidar", 1.)]
            pose = state.x, state.y, state.yaw
            self.executor.last_detections = [(d, pose) for d in detections]
            self.executor._remember(detections, pose)
            return {}
        self.executor.look = look
        result = self.call("approach", {"target": "laptop"})
        self.assertEqual(result["status"], "queued", result)
        self.assertTrue(wait_for(lambda: self.events, 25))
        self.assertEqual(self.events[-1].status, "completed", self.events[-1].detail)
        self.assertAlmostEqual(abs(turns[0]), math.pi, delta=.1)
        self.assertAlmostEqual(self.events[-1].detail["final_gap_m"], .5, delta=.2)


if __name__ == "__main__":
    unittest.main()
