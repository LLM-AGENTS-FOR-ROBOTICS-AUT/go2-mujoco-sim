"""Offline, headless MuJoCo adapter for coverage evaluation in simulated time.

The same SimGo2 policy, physics, L1 rays and Executor safety run in-process.
Only the executor's time source is temporarily replaced by SimulationClock in
the standalone evaluator process. No physical-robot adapter uses this clock.
"""

import threading
import time

from go2.adapters.base import AdapterError, RobotAdapter
from go2.adapters.sim import SimAdapter
from robot import SimGo2
from sensors import Lidar


class SimulationClock:
    def __init__(self, adapter):
        self.adapter = adapter

    def monotonic(self):
        with self.adapter.sim.lock:
            return float(self.adapter.sim.data.time)

    def time(self):
        return time.time()

    def sleep(self, seconds):
        target = self.monotonic() + seconds
        # Synchronous stepping: planning/scan computation does not let physics
        # run ahead of the control loop. Every motion still uses actual policy
        # and physics steps, with the same simulated sensing/control intervals.
        while self.monotonic() < target:
            if self.adapter.closed.is_set():
                raise AdapterError("offline simulator stopped")
            with self.adapter.sim.lock:
                now = float(self.adapter.sim.data.time)
                if self.adapter.last_move is not None and now - self.adapter.last_move > .5:
                    self.adapter.sim.StopMove()
                    self.adapter.last_move = None
                self.adapter.sim.step()


class SteppedSimAdapter(RobotAdapter):
    kind, label = "sim", "Offline MuJoCo exploration evaluation"
    scene = "house"

    def __init__(self, world):
        self.sim = SimGo2("house", world)
        self.sensor = Lidar(self.sim.model)
        self.closed = threading.Event()
        self.clock = SimulationClock(self)
        self.active, self.error, self.last_move = False, None, None

    def connect(self):
        self.active = True

    @property
    def connected(self):
        return self.active and not self.closed.is_set()

    def close(self):
        self.closed.set()
        self.active = False
        self.sim.StopMove()

    def state(self):
        if self.error:
            raise AdapterError(self.error)
        return SimAdapter._state(self.sim.state())

    def move(self, vx, vy, wz):
        with self.sim.lock:
            result = self.sim.Move(vx, vy, wz)
            if not result.get("ok"):
                raise AdapterError(result.get("error", "move refused"))
            self.last_move = float(self.sim.data.time)

    def stop_move(self):
        with self.sim.lock:
            self.last_move = None
            self.sim.StopMove()

    def sport(self, name, **args):
        if name not in ("stand", "stand_up"):
            raise AdapterError("offline exploration evaluation only needs standing")
        result = self.sim.BalanceStand() if name == "stand" else self.sim.StandUp()
        if not result.get("ok"):
            raise AdapterError(result.get("error", "stand refused"))

    def lidar(self):
        with self.sim.lock:
            return self.sensor.scan(self.sim.data).astype(float), self.state()

    def camera(self):
        raise AdapterError("coverage evaluation uses LiDAR; RGB search is evaluated separately")
