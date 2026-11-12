import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from go2.adapters.base import CameraFrame, RobotState
from go2.memory_visibility import clear_saved_location
from go2.object_memory import ObjectMemory
from go2.executor import Executor
from go2.agent import Agent
from go2.mock_model import MockModel
from tests.fakes import FakeRobot, wait_for


class MissingMemoryTests(unittest.TestCase):
    def test_requires_three_separate_clear_views_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"memory.json"
            memory = ObjectMemory(path)
            item = memory.remember("laptop", 2., 0.)
            for stamp in (0., .05, .2):
                self.assertEqual(memory.observe_empty([item["id"]], observed_at=stamp), [])
            self.assertEqual(len(memory.recall("laptop")), 1)
            changed = memory.observe_empty([item["id"]], observed_at=.4)
            self.assertEqual(changed[0]["state"], "missing")
            restarted = ObjectMemory(path)
            self.assertEqual(restarted.recall("laptop"), [])
            self.assertEqual(restarted.snapshot()[0]["id"], item["id"])
            restored = restarted.remember("laptop", 2.1, 0.)
            self.assertEqual(restored["id"], item["id"])
            self.assertEqual(restored["times_seen"], 2)
            self.assertEqual(restored["state"], "active")

    def test_occluded_seen_or_reconnected_resets_pending_evidence(self):
        for reset in ("occluded", "seen", "reconnected"):
            with self.subTest(reset=reset):
                m = ObjectMemory()
                key = m.remember("chair", 1., 0.)["id"]
                m.observe_empty([key], observed_at=0.)
                m.observe_empty([key], observed_at=.2)
                if reset == "reconnected":
                    m.reset_observation_evidence()
                else:
                    m.observe_empty([] if reset == "occluded" else [key], observed_at=.4,
                                    seen_ids=[key] if reset == "seen" else [])
                self.assertEqual(m.observe_empty([key], observed_at=.6), [])
                self.assertEqual(len(m.recall("chair")), 1)

    def test_far_lookalike_keeps_both_identities(self):
        m = ObjectMemory()
        a = m.remember("chair", 1., 0., "blue")
        b = m.remember("chair", 3., 0., "blue")
        self.assertNotEqual(a["id"], b["id"])
        self.assertEqual(len(m.recall("chair")), 2)

    def test_explicit_forgetting_is_persisted_without_removing_other_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"memory.json"
            m = ObjectMemory(path)
            a = m.remember("chair", 1., 0.)
            b = m.remember("chair", 3., 0.)
            m.forget(a["id"])
            self.assertEqual([r["id"] for r in ObjectMemory(path).snapshot()], [b["id"]])
            with self.assertRaises(ValueError):
                m.forget(a["id"])

    def test_legacy_file_migrates_and_invalid_rows_do_not_poison_queries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"memory.json"
            path.write_text(json.dumps([{"label": "chair", "x": 1., "y": 2.},
                                        {"label": "bad", "x": "NaN", "y": 2.}]))
            m = ObjectMemory(path)
            self.assertEqual(len(m.snapshot()), 1)
            self.assertIsNotNone(m.load_error)
            m.save()
            self.assertEqual(m.snapshot(), ObjectMemory(path).snapshot())


class EmptyLocationVisibilityTests(unittest.TestCase):
    def setUp(self):
        state = RobotState(time.monotonic(), 0., 0., 0., posture="standing")
        # Camera optical z is robot forward, optical x is robot right.
        self.frame = CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), [],
                                 dict(fx=250., fy=250., cx=320., cy=240.),
                                 np.array([0., 0., 0.]),
                                 np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]), state)
        self.item = dict(x=2., y=0.)
        x, y = np.meshgrid(np.linspace(1.92, 2.08, 4), np.linspace(-.08, .08, 4))
        self.floor = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])

    def test_direct_empty_floor_is_positive_evidence(self):
        self.assertTrue(clear_saved_location(self.item, self.frame, self.floor))

    def test_obstacles_occlusion_sparse_stale_and_motion_are_not_evidence(self):
        for cloud in (self.floor[:2], np.vstack([self.floor, [2., 0., .2]]),
                      np.vstack([self.floor, [1., 0., .2]])):
            self.assertFalse(clear_saved_location(self.item, self.frame, cloud))
        self.assertFalse(clear_saved_location(dict(x=-2., y=0.), self.frame, self.floor))
        self.frame.state.stamp -= 1.
        self.assertFalse(clear_saved_location(self.item, self.frame, self.floor))
        self.frame.state.stamp = time.monotonic()
        self.frame.state.vx = .2
        self.assertFalse(clear_saved_location(self.item, self.frame, self.floor))
        self.frame.state.vx = 0.
        self.frame.state.extra["localization"] = {"state": "lost", "accepted": False}
        self.assertFalse(clear_saved_location(self.item, self.frame, self.floor))


class MissingObjectArrivalTests(unittest.TestCase):
    def test_mock_reports_verified_empty_location_when_approach_arrives(self):
        robot = FakeRobot(camera=True)
        robot.kind = "sim"
        robot.connect()
        self.addCleanup(robot.close)
        robot.camera = lambda: CameraFrame(np.zeros((480, 640, 3), dtype=np.uint8), [],
            dict(fx=250., fy=250., cx=320., cy=240.), np.zeros(3),
            np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]]), robot.state())
        executor = Executor()
        executor.attach(robot)
        self.addCleanup(executor.close)
        executor.object_memory.remember("laptop", .9, 0.)
        agent = Agent(MockModel(), executor)
        executor.on_event = agent.on_task_event
        self.addCleanup(agent.close)
        agent.robot_line = lambda: "standing in the room"
        agent.submit("go to the laptop")
        self.assertTrue(wait_for(lambda: not agent.busy, 8))
        self.assertIn("may have moved", agent.messages[-1]["content"])
        self.assertEqual(executor.object_memory.snapshot()[0]["state"], "missing")
        self.assertIsNone(executor._recall("laptop"))
        self.assertFalse(any(c[0] == "move" for c in robot.calls))
