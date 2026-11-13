"""LiDAR buffer timestamps share the executor/follow tracker's clock domain."""
from dataclasses import replace
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from go2 import executor as X, follow as F, perception as P
from tests.fakes import FakeRobot


class ScanClockTests(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobot(camera=True)
        self.robot.connect()
        self.addCleanup(self.robot.close)
        original = self.robot.state
        self.robot.state = lambda: replace(original(), extra={"sim_time": 12.})
        self.executor = X.Executor()
        self.executor.attach(self.robot)
        self.addCleanup(self.executor.close)

    def test_simulator_scan_uses_executor_clock_for_following(self):
        now = time.monotonic()
        clock = SimpleNamespace(monotonic=lambda: now, time=time.time, sleep=time.sleep)
        feed = F.PersonFeed(self.executor, self.robot)
        feed.tracker = F.PersonTracker(2., 0., now, "red")
        camera = self.robot.camera
        def one_frame():
            frame = camera()
            feed.stop()
            return frame
        self.robot.camera = one_frame
        person = P.Detection("person1", "person", None, 2., 0., 2., 0., "lidar", 1., clothing="red")
        with patch.object(X, "time", clock), patch.object(P, "fuse", return_value=[person]):
            self.executor._refresh_cloud()
            feed.run()
        self.assertEqual(self.executor.cloud.scans[-1][0], now)
        self.assertLess(time.monotonic()-feed.tracker.seen, F.SEEN_S)

    def test_offline_evaluator_can_supply_its_own_clock(self):
        clock = SimpleNamespace(monotonic=lambda: 123., time=time.time, sleep=time.sleep)
        with patch.object(X, "time", clock):
            self.executor._refresh_cloud()
        self.assertEqual(self.executor.cloud.scans[-1][0], 123.)
