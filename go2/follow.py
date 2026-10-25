"""Follow-me (#15): keep a chosen person in view and walk after them, through the executor's safety checks.

Three parts, run by Executor._follow:
  PersonFeed     a camera thread: segment + LiDAR-fuse each frame, people's positions in odom (~5 Hz in software
                 rendering, faster with a GPU);
  PersonTracker  a constant-velocity (alpha-beta) tracker locked onto one person. It never switches: a person far
                 from where the target should be is ignored, a clearly different shirt colour is ignored, and when
                 two candidates are about equally likely it holds still until they separate;
  follow_step    a distance and heading controller. Its forward speed is capped so the LiDAR stopping margin
                 always holds (the person's own legs are in that corridor, so it can never walk into them).

Simulator only for now: the real Go2's own follow and avoidance modes would need switching off (see the issue)."""
from __future__ import annotations

import math
import threading
import time

from . import limits as L
from . import perception as P
from .adapters.base import AdapterError

GATE_M = .7               # how far from the predicted position a detection may be and still be the target
AMBIGUOUS_M = .45         # two candidates closer than this to equally likely: don't guess
ALPHA, BETA = .6, .25     # tracker gains (position, velocity)
MAX_PERSON_SPEED = 2.     # m/s, caps the velocity estimate
PREDICT_MAX_S = 1.        # never extrapolate further than this
SEEN_S = .8               # target counts as in view if seen this recently (a few camera frames)
LOST_STOP_S = 2.          # after this long unseen: stop walking, turn to where they were, look
GIVE_UP_S = 8.            # after this long unseen: end and tell the user
BLOCKED_GIVE_UP_S = 3.    # held up by an obstacle (not the person) this long: end as blocked
TICK_S = .1
COLOUR_FAMILY = {"red": {"orange", "pink", "brown"}, "orange": {"red", "brown", "yellow"},
                 "brown": {"orange", "red", "black"}, "yellow": {"orange", "green"}, "green": {"yellow", "cyan"},
                 "cyan": {"green", "blue"}, "blue": {"cyan", "purple"}, "purple": {"blue", "pink"},
                 "pink": {"red", "purple"}, "white": {"grey"}, "grey": {"white", "black"}, "black": {"grey", "brown"}}


def colour_fits(seen, wanted):
    """Does a seen shirt colour fit the wanted one? Unknown counts as fitting (lighting changes colours)."""
    return seen is None or wanted is None or seen == wanted or seen in COLOUR_FAMILY.get(wanted, ())


def colour_conflicts(seen, wanted):
    return seen is not None and wanted is not None and not colour_fits(seen, wanted)


class PersonTracker:
    """One person in odom coordinates. update() takes everyone seen in a frame and returns the index of the
    target among them, or None (not seen, or not sure which one it is)."""

    def __init__(self, x, y, t, clothing=None):
        self.x, self.y, self.vx, self.vy = x, y, 0., 0.
        self.t = self.seen = t
        self.clothing = clothing
        self.ambiguous = False
        self.frames = 1

    def predict(self, t):
        h = min(max(t - self.t, 0.), PREDICT_MAX_S)
        return self.x + self.vx * h, self.y + self.vy * h

    def update(self, t, people):
        """people: [(x, y, clothing)] in odom."""
        px, py = self.predict(t)
        gate = GATE_M + min(1.5, .6 * max(0., t - self.seen))
        near = sorted((math.hypot(x - px, y - py), i) for i, (x, y, c) in enumerate(people)
                      if math.hypot(x - px, y - py) < gate and not colour_conflicts(c, self.clothing))
        if not near:
            self.ambiguous = False
            return None
        if len(near) > 1 and near[1][0] - near[0][0] < AMBIGUOUS_M:
            exact = [n for n in near if people[n[1]][2] == self.clothing and self.clothing is not None]
            if len(exact) != 1:
                self.ambiguous = True
                return None
            near = exact
        _, i = near[0]
        x, y, clothing = people[i]
        dt = max(t - self.t, 1e-3)
        rx, ry = x - px, y - py
        self.x, self.y = px + ALPHA * rx, py + ALPHA * ry
        self.vx, self.vy = self.vx + BETA * rx / dt, self.vy + BETA * ry / dt
        speed = math.hypot(self.vx, self.vy)
        if speed > MAX_PERSON_SPEED:
            self.vx, self.vy = self.vx * MAX_PERSON_SPEED / speed, self.vy * MAX_PERSON_SPEED / speed
        self.t = self.seen = t
        self.ambiguous = False
        self.frames += 1
        if self.clothing is None and clothing is not None:
            self.clothing = clothing
        return i


class PersonFeed(threading.Thread):
    """Grabs camera frames while following and hands the people in them to the tracker (once there is one)."""

    def __init__(self, executor, adapter):
        super().__init__(daemon=True, name="follow-camera")
        self.executor, self.adapter = executor, adapter
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.frames = 0
        self.people = []          # latest frame: [(x, y, clothing, Detection)] in odom
        self.stamp = 0.
        self.error = None
        self.tracker = None

    def stop(self):
        self.done.set()

    def run(self):
        ex = self.executor
        while not self.done.is_set():
            try:
                frame = self.adapter.camera()
            except AdapterError as error:
                self.error = str(error)
                if self.done.wait(.3):
                    return
                continue
            except Exception as error:   # never kill following silently
                self.error = f"camera: {error}"
                return
            s = frame.state
            pose = (s.x, s.y, s.yaw)
            with ex._cloud_lock:
                local = ex.cloud.level(pose)
            cloud_base = local[:0]
            if len(local):
                local[:, 2] -= s.height
                cloud_base = local @ P.rotation_rp(s.roll, s.pitch)
            detections = P.fuse(frame.masks, frame.rgb, cloud_base, frame.intrinsics, frame.camera_pos,
                                frame.camera_rot, s.roll, s.pitch, s.height)
            if ex.detection_filter is not None:
                detections = ex.detection_filter(detections)
            c, sn = math.cos(s.yaw), math.sin(s.yaw)
            people = [(s.x + c * d.x_m - sn * d.y_m, s.y + sn * d.x_m + c * d.y_m, d.clothing, d)
                      for d in detections if d.label == "person" and d.x_m is not None]
            now = time.monotonic()
            tracked = None
            with self.lock:
                self.people, self.stamp, self.error = people, now, None
                self.frames += 1
                if self.tracker is not None:
                    i = self.tracker.update(now, [p[:3] for p in people])
                    tracked = None if i is None else people[i][3]
            boxes = [{"label": d.label, "score": round(d.confidence, 2), "box": d.box,
                      **({"tracked": True} if d is tracked else {})} for d in detections if d.box]
            with ex.lock:
                ex._stable_ids(detections, pose)
                ex.last_detections = [(d, pose) for d in detections]
                ex._remember(detections, pose)
                ex.last_frame = frame.rgb
                ex.last_frame_meta = {"stamp": time.time(), "objects": len(detections), "boxes": boxes}

    def wait_frames(self, count, timeout):
        """Wait for `count` more frames; False on timeout."""
        with self.lock:
            target = self.frames + count
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.done.is_set():
            with self.lock:
                if self.frames >= target:
                    return True
            time.sleep(.02)
        return False


def max_safe_speed(clear):
    """Fastest speed whose stopping margin (Executor._margin) fits in `clear` metres."""
    a, b, c = 1 / 2.4, .25, .22 - clear
    if c >= 0:
        return 0.
    return min(L.MAX_SPEED_MPS, (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a))


def describe_person(clothing, distance, bearing):
    who = f"the person in {clothing}" if clothing else "a person"
    return f"{who} {distance:.1f} m {P.describe_bearing(bearing)}"


def follow_step(gap, bearing, set_gap, target_speed_along):
    """Velocity command (vx, wz) to close the gap to set_gap while facing the person."""
    vx = max(-.25, min(L.MAX_SPEED_MPS, .8 * target_speed_along + .9 * (gap - set_gap)))
    if abs(gap - set_gap) < .1 and abs(target_speed_along) < .1:
        vx = 0.
    wz = max(-L.MAX_TURN_RATE_RPS, min(L.MAX_TURN_RATE_RPS, 2. * bearing))
    if abs(bearing) > math.radians(45):
        vx = min(vx, .15)          # mostly turn first
    return vx, wz
