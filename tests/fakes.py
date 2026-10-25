"""Test doubles: a kinematic robot with box obstacles, and a scripted chat model; and a check that skips camera
tests where MuJoCo can't render."""
from __future__ import annotations

import functools
import math
import subprocess
import sys
import threading
import time

import numpy as np

from go2.adapters.base import AdapterError, CameraFrame, RobotAdapter, RobotState


class FakeRobot(RobotAdapter):
    """Integrates commanded body velocities about 100 times a second, over the time that actually passed, with
    acceleration limits and a 0.5 s dead-man."""
    kind, label = "fake", "Fake robot"

    def __init__(self, obstacles=(), requires_arming=False, unsupported=(), posture="standing", height=.3,
                 blind_behind=False, drops=(), glass=(), people=(), camera=False):
        self.blind_behind = blind_behind
        self.drops = list(drops)           # (xmin, xmax, ymin, ymax) boxes in odom where the floor is 0.5 m lower
        self.glass = list(glass)           # boxes that stop the robot but return nothing to the LiDAR
        self.contact_at = None             # when the body first touched glass
        self.stop_times = []
        self.paused_until = 0.             # simulate a busy computer: the world stops advancing, readings don't change
        self.start_lag = 0.                # legs take this long to get going after a move from standstill
        self._moving_since = None
        self.people = list(people)         # FakePerson: walk paths, seen by the LiDAR and (patched fuse) the camera
        self.has_camera = camera
        self.requires_arming = requires_arming
        self.obstacles = list(obstacles)   # (xmin, xmax, ymin, ymax) boxes in odom, 0.6 m tall
        self.unsupported = set(unsupported)
        self.x = self.y = self.yaw = 0.
        self.v = np.zeros(3)
        self.cmd = np.zeros(3)
        self.posture = posture
        self.height = height
        self.busy_until = 0.
        self.last_move = None
        self.calls = []
        self.freeze = False          # stop publishing fresh telemetry
        self.lock = threading.RLock()
        self._closed = threading.Event()
        self._connected = False
        self.frozen_stamp = None

    def connect(self):
        self._connected = True
        threading.Thread(target=self._loop, daemon=True).start()

    def close(self):
        self._connected = False
        self._closed.set()

    @property
    def connected(self):
        return self._connected

    def _loop(self):
        last = time.monotonic()
        while not self._closed.wait(.01):
            now = time.monotonic()
            dt, last = min(now - last, .1), now   # the time that really passed: a busy machine wakes this late
            if now < self.paused_until:
                continue
            with self.lock:
                if self.last_move is not None and time.monotonic() - self.last_move > .5:
                    self.cmd[:] = 0
                    self.last_move = None
                step = np.array([1.6, 1.6, 4.0]) * dt
                if self.cmd.any() and self._moving_since is None:
                    self._moving_since = now
                elif not self.cmd.any():
                    self._moving_since = None
                lagging = self._moving_since is not None and now - self._moving_since < self.start_lag
                if not lagging:
                    self.v += np.clip(self.cmd - self.v, -step, step)
                c, s = math.cos(self.yaw), math.sin(self.yaw)
                nx = self.x + (c * self.v[0] - s * self.v[1]) * dt
                ny = self.y + (s * self.v[0] + c * self.v[1]) * dt
                if self._touches_glass(nx, ny):
                    self.v[:2] = 0                 # pressed against it: no progress, like a real stall
                    if self.contact_at is None:
                        self.contact_at = time.monotonic()
                else:
                    self.x, self.y = nx, ny
                self.yaw = math.atan2(math.sin(self.yaw + self.v[2] * dt), math.cos(self.yaw + self.v[2] * dt))
                if self.posture == "walking" and not self.cmd.any() and np.abs(self.v).max() < 1e-3:
                    self.posture = "standing"
                if self.posture == "busy" and time.monotonic() > self.busy_until:
                    self.posture = self._after_busy

    def state(self):
        with self.lock:
            if self.freeze:
                if self.frozen_stamp is None:
                    self.frozen_stamp = time.monotonic()
                stamp = self.frozen_stamp
            else:
                stamp = time.monotonic()
            return RobotState(stamp=stamp, x=self.x, y=self.y, yaw=self.yaw, vx=self.v[0], vy=self.v[1],
                              wz=self.v[2], height=self.height, posture=self.posture)

    def move(self, vx, vy, wz):
        with self.lock:
            self.calls.append(("move", round(vx, 3), round(vy, 3), round(wz, 3)))
            if self.posture not in ("standing", "walking"):
                raise AdapterError(f"not allowed while {self.posture}")
            self.cmd[:] = (vx, vy, wz)
            self.last_move = time.monotonic()
            if any(self.cmd):
                self.posture = "walking"

    def stop_move(self):
        with self.lock:
            self.calls.append(("stop_move",))
            self.stop_times.append(time.monotonic())
            self.cmd[:] = 0
            self.last_move = None

    def supports(self, name):
        return f"{name} is not supported here" if name in self.unsupported else None

    def sport(self, name, **args):
        with self.lock:
            self.calls.append(("sport", name, args))
            targets = {"stand": "standing", "stand_up": "locked_stand", "lie_down": "lying", "sit": "sitting",
                       "recovery_stand": "locked_stand", "damp": "damp"}
            self._after_busy = targets.get(name, "standing")
            self.posture = "busy"
            self.busy_until = time.monotonic() + .2

    def lidar(self):
        """Points on obstacle faces (0.1-0.5 m high) plus a floor ring, in the base frame."""
        with self.lock:
            pts = []
            now = time.monotonic()
            people = [(x - .15, x + .15, y - .15, y + .15) for x, y in
                      (p.at(now) for p in self.people) if x is not None]
            for xmin, xmax, ymin, ymax in self.obstacles + people:
                for z in (.1, .3, .5):
                    for t in np.linspace(0, 1, 25):
                        for px, py in ((xmin + t * (xmax - xmin), ymin), (xmin + t * (xmax - xmin), ymax),
                                       (xmin, ymin + t * (ymax - ymin)), (xmax, ymin + t * (ymax - ymin))):
                            pts.append((px, py, z))
            # Floor grid within 3 m, minus the shadows cast by obstacles (like real LiDAR returns).
            g = np.arange(-3, 3.01, .08)
            gx, gy = np.meshgrid(g, g)
            keep = gx ** 2 + gy ** 2 <= 9
            floor = np.stack([self.x + gx[keep], self.y + gy[keep]], 1)
            floor = floor[~self._shadowed(floor)]
            def floor_z(fx, fy):
                return -.5 if any(a <= fx <= b and c <= fy <= d for a, b, c, d in self.drops) else 0.
            pts = np.array(pts + [(fx, fy, floor_z(fx, fy)) for fx, fy in floor], dtype=float)
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            dx, dy = pts[:, 0] - self.x, pts[:, 1] - self.y
            base = np.stack([c * dx + s * dy, -s * dx + c * dy, pts[:, 2] - self.height], 1)
            if self.blind_behind:
                base = base[base[:, 0] > -.3]
        return base, self.state()

    def _touches_glass(self, x, y, radius=.36):
        for xmin, xmax, ymin, ymax in self.glass:
            dx = max(xmin - x, 0, x - xmax)
            dy = max(ymin - y, 0, y - ymax)
            if math.hypot(dx, dy) < radius:
                return True
        return False

    def _shadowed(self, floor):
        """Slab test: does the sight line from the robot to each floor point cross an obstacle box?"""
        hidden = np.zeros(len(floor), bool)
        origin = np.array([self.x, self.y])
        d = floor - origin
        with np.errstate(divide="ignore", invalid="ignore"):
            for xmin, xmax, ymin, ymax in self.obstacles:
                lo = np.array([xmin, ymin])
                hi = np.array([xmax, ymax])
                t1, t2 = (lo - origin) / d, (hi - origin) / d
                tmin = np.nanmax(np.minimum(t1, t2), axis=1)
                tmax = np.nanmin(np.maximum(t1, t2), axis=1)
                hidden |= (tmax >= np.maximum(tmin, 0)) & (tmin <= 1)
        return hidden

    def camera(self):
        if not self.has_camera:
            raise AdapterError("fake robot has no camera")
        time.sleep(.05)                        # a frame takes a while, as on the real thing
        return CameraFrame(rgb=np.zeros((4, 4, 3), np.uint8), masks=[], intrinsics={}, camera_pos=np.zeros(3),
                           camera_rot=np.eye(3), state=self.state())

    def seen_people(self, hfov_deg=120., max_m=6.):
        """What a camera + LiDAR fusion would report: Detections of the people in front, in the level frame
        (used in place of perception.fuse in tests; FakeRobot renders no images)."""
        from go2.perception import Detection
        s, now, out = self.state(), time.monotonic(), []
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        for i, person in enumerate(self.people):
            px, py = person.at(now)
            if px is None:
                continue
            dx, dy = c * (px - s.x) + sn * (py - s.y), -sn * (px - s.x) + c * (py - s.y)
            r = math.hypot(dx, dy)
            bearing = math.degrees(math.atan2(dy, dx))
            if abs(bearing) > hfov_deg / 2 or r > max_m or r < .2:
                continue
            nx, ny = dx * (r - .15) / r, dy * (r - .15) / r          # near surface
            out.append(Detection(id=f"obj{i + 1}", label="person", color=None, distance_m=r - .15, bearing_deg=bearing,
                                 x_m=nx, y_m=ny, distance_source="lidar", confidence=.9, clothing=person.colour,
                                 box=[.4, .2, .6, .9]))
        return out


class FakePerson:
    """Walks a polyline at `speed` m/s from when it is created (or stands still with one point); hide() removes them."""

    def __init__(self, path, speed=.5, colour="red", delay=0.):
        self.path, self.speed, self.colour = [tuple(map(float, p)) for p in path], speed, colour
        self.start, self.hidden = time.monotonic() + delay, False

    def hide(self):
        self.hidden = True

    def at(self, now):
        if self.hidden:
            return None, None
        travel = max(0., now - self.start) * self.speed
        for (x0, y0), (x1, y1) in zip(self.path, self.path[1:]):
            leg = math.hypot(x1 - x0, y1 - y0)
            if travel <= leg:
                f = travel / leg if leg else 0.
                return x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
            travel -= leg
        return self.path[-1]


class ScriptedModel:
    """Returns queued responses; a response may be a callable(messages) -> response."""
    model_id = "scripted"
    options = {}
    loaded = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.cancelled = 0

    def check(self):
        return {"model": "scripted", "ready": True}

    def chat(self, messages, tools=None, cancelled=lambda: False, on_text=None, timeout=60):
        self.requests.append([dict(m) for m in messages])
        if not self.responses:
            return {"content": "", "tool_calls": [], "stats": {}}
        response = self.responses.pop(0)
        if callable(response):
            response = response(messages)
        if isinstance(response, str):
            return {"content": response, "tool_calls": [], "stats": {}}
        content, calls = response
        return {"content": content, "tool_calls": [{"name": n, "arguments": a} for n, a in calls], "stats": {}}

    def cancel(self):
        self.cancelled += 1

    def unload(self):
        return {"unloaded": True}


def wait_for(predicate, timeout=10.0, interval=.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


RENDER_PROBE = ("import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco/>'), 16, 16); "
                "r.render(); r.close()")


@functools.cache
def render_problem():
    """Why MuJoCo can't render offscreen on this machine, or None. GitHub's hosted runners have no GPU, so the
    simulated camera can't render there. Checked in a separate process, like the simulator's own rendering."""
    try:
        done = subprocess.run([sys.executable, "-c", RENDER_PROBE], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"MuJoCo can't render offscreen here ({error})"
    if done.returncode:
        lines = done.stderr.strip().splitlines() or [f"exit code {done.returncode}"]
        return f"MuJoCo can't render offscreen here ({lines[-1]})"
    return None


def needs_camera(test):
    """Skip a test that uses the simulated camera where MuJoCo can't render, instead of failing it."""
    @functools.wraps(test)
    def run(self, *args, **kwargs):
        problem = render_problem()
        if problem:
            self.skipTest(problem)
        return test(self, *args, **kwargs)
    return run
