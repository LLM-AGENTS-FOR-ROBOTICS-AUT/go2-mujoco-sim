"""Normal process/RPC integration, alongside the offline coverage benchmark."""
import math
import sys
import tempfile
import unittest

try:
    import mujoco
    import torch
    HAVE_SIM = True
except ImportError:
    HAVE_SIM = False

from go2.executor import Executor
from go2.adapters.sim import SimAdapter
from tests.fakes import wait_for


@unittest.skipUnless(HAVE_SIM, "MuJoCo and walking dependencies not installed")
class ExplorationSimulationTests(unittest.TestCase):
    def test_world_viewpoint_uses_normal_adapter_and_finishes_without_contact(self):
        sim = SimAdapter(python=sys.executable, scene="empty", headless=True)
        sim.connect()
        self.addCleanup(sim.close)
        with tempfile.TemporaryDirectory() as temp:
            events = []
            executor = Executor(on_event=events.append, memory_path=temp + "/memory.json")
            self.addCleanup(executor.close)
            executor.attach(sim)
            start = sim.state()
            goal = (start.x + .7, start.y + .2, .25)
            result = executor.submit_viewpoint(goal, timeout_s=40)
            self.assertEqual(result["status"], "queued", result)
            self.assertTrue(wait_for(lambda: events, 45), "viewpoint did not finish")
            self.assertEqual(events[-1].status, "completed", events[-1].detail)
            state = sim.state()
            self.assertLess(math.hypot(state.x - goal[0], state.y - goal[1]), .18)
            self.assertAlmostEqual(state.yaw, goal[2], delta=.15)
            self.assertEqual(state.extra["collision_count"], 0)
            self.assertIsNone(state.fault)
            executor.close()


if __name__ == "__main__":
    unittest.main()
